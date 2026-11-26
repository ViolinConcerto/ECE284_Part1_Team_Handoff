`timescale 1ns/1ps
module rtl_units_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg [3:0] a = 0, b = 0;
    reg signed [31:0] c = 0;
    reg signed_mode = 0;
    wire signed [31:0] mac_out;
    mac arithmetic(.a(a), .b(b), .c(c), .act_signed(signed_mode), .out(mac_out));

    reg reset = 1, compute_clear = 0;
    reg [3:0] pe_input = 0;
    reg [1:0] pe_instruction = 0;
    reg pe_signed = 0;
    wire signed [31:0] pe_output;
    wire [3:0] pe_east;
    wire [1:0] pe_instruction_east;
    wire pe_signed_east;
    mac_tile pe(.clk(clk), .reset(reset), .compute_clear(compute_clear),
        .in_w(pe_input), .out_e(pe_east), .in_n(32'b0), .out_s(pe_output),
        .inst_w(pe_instruction), .inst_e(pe_instruction_east),
        .act_signed(pe_signed), .act_signed_e(pe_signed_east));

    reg clear = 0, acc = 0, relu = 0;
    reg [4:0] shift = 0;
    reg signed [31:0] sfp_input = 0;
    wire signed [31:0] sfp_output;
    sfp special(.clk(clk), .reset(reset), .clear(clear), .acc(acc),
        .relu(relu), .shift(shift), .in(sfp_input), .out(sfp_output));

    integer mode, av, bv, ai, bi, expected;
    task tick;
        begin @(posedge clk); #1; @(negedge clk); end
    endtask
    task expect_sfp(input signed [31:0] wanted);
        begin
            if (sfp_output !== wanted)
                $fatal(1, "SFP got=%0d expected=%0d", sfp_output, wanted);
        end
    endtask
    initial begin
        for (mode=0; mode<2; mode=mode+1)
            for (av=0; av<16; av=av+1)
                for (bv=0; bv<16; bv=bv+1) begin
                    a=av; b=bv; signed_mode=mode;
                    ai=(mode && av>=8) ? av-16 : av;
                    bi=bv>=8 ? bv-16 : bv;
                    c=-100000;
                    expected=ai*bi-100000;
                    #1;
                    if (mac_out !== expected)
                        $fatal(1, "MAC mode=%0d a=%0d b=%0d got=%0d expected=%0d",
                               mode, ai, bi, mac_out, expected);
                end
        @(negedge clk);
        tick(); reset=0;
        pe_input=4'b1011; pe_instruction=1; tick();
        pe_instruction=0; tick();
        if (pe.b_q !== 4'b1011 || pe.load_ready_q !== 0)
            $fatal(1, "PE did not retain loaded weight");
        compute_clear=1; tick(); compute_clear=0;
        if (pe.b_q !== 4'b1011 || pe.load_ready_q !== 0)
            $fatal(1, "compute_clear changed the weight or loaded flag");
        pe_input=14; pe_instruction=2; pe_signed=1; tick();
        if (pe_output !== 10) $fatal(1, "signed delta PE failed");
        pe_signed=0; tick();
        if (pe_output !== -70) $fatal(1, "unsigned nibble PE failed");
        pe_instruction=0;

        clear=1; tick(); clear=0;
        sfp_input=-7; shift=4; acc=1; tick(); acc=0;
        expect_sfp(-112);
        sfp_input=999; shift=0; tick(); expect_sfp(-112);
        sfp_input=3; acc=1; tick(); acc=0; expect_sfp(-109);
        clear=1; tick(); clear=0; expect_sfp(0);
        sfp_input=100; acc=1; tick();
        sfp_input=-17; tick(); acc=0; expect_sfp(83);
        clear=1; tick(); clear=0;
        sfp_input=-42; acc=1; tick(); acc=0;
        relu=1; tick(); relu=0; expect_sfp(0);
        clear=1; tick(); clear=0;
        sfp_input=-100000; shift=4; acc=1; tick(); acc=0;
        expect_sfp(-1600000);
        $display("PASS video RTL units: 512 MAC combinations, weight retention, signed PE, SFU shift/hold/reconstruct");
        $finish;
    end
    initial begin #20000; $fatal(1, "unit timeout"); end
endmodule
